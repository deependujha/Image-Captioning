'use client';
import Image_Uploader from '@/components/Image_Uploader';
import MainContainer from '@/components/MainContainer';
import { ThreeDCardDemo } from '@/components/My3dCard';
import My_Image_Caption_Card from '@/components/My_Image_Caption_Card';
import { BackgroundGradientAnimation } from '@/components/ui/background-gradient-animation';

const Page = () => {
	const btnClick = () => {
		console.log('hello world');
	};
	return (
		<div>
			{/* <BackgroundGradientAnimation interactive={false}> */}
			{/* <Image_Uploader /> */}
			{/* <MainContainer /> */}
			{/* <ThreeDCardDemo /> */}
			{/* <button onClick={btnClick}>click me</button> */}
			{/* </BackgroundGradientAnimation> */}
			<My_Image_Caption_Card />
		</div>
	);
};

export default Page;
